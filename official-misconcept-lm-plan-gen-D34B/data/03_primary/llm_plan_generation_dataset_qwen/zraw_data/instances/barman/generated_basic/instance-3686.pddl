(define (problem prob)
 (:domain barman)
 (:objects 
      shaker160 - shaker
      left right - hand
      shot111 shot429 shot241 - shot
      ingredient175 ingredient160 ingredient103 ingredient249 - ingredient
      cocktail206 - cocktail
      dispenser383 dispenser308 dispenser232 dispenser270 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker160)
  (ontable shot111)
  (ontable shot429)
  (ontable shot241)
  (dispenses dispenser383 ingredient175)
  (dispenses dispenser308 ingredient160)
  (dispenses dispenser232 ingredient103)
  (dispenses dispenser270 ingredient249)
  (clean shaker160)
  (clean shot111)
  (clean shot429)
  (clean shot241)
  (empty shaker160)
  (empty shot111)
  (empty shot429)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker160 l0)
  (shaker-level shaker160 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient103)
  (cocktail-part2 cocktail206 ingredient175)
)
 (:goal
  (and
      (contains shot111 cocktail206)
      (contains shot429 cocktail206)
)))
