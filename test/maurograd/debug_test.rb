require 'maurograd/tensor'
require 'maurograd/utils/utils'

describe "Maurograd.debug" do
  it "is off by default" do
    expect(Maurograd.debug).to be == false
  end

  it "gates Utils.assert_finite! so it only raises when enabled" do
    original = Maurograd.debug
    begin
      bad = Numo::SFloat[Float::NAN, 1.0]

      Maurograd.debug = false
      expect { Maurograd::Utils.assert_finite!(bad, where: "test") }.not.to raise_exception

      Maurograd.debug = true
      expect { Maurograd::Utils.assert_finite!(bad, where: "test") }.to raise_exception
    ensure
      Maurograd.debug = original
    end
  end

  it "can be enabled via the MAUROGRAD_DEBUG=1 environment variable at load time" do
    out = `MAUROGRAD_DEBUG=1 ruby -Ilib -e 'require "maurograd"; puts Maurograd.debug'`
    expect(out.strip).to be == "true"
  end
end
