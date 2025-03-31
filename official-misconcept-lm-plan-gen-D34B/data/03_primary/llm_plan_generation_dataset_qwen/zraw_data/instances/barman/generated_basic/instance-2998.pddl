(define (problem prob)
 (:domain barman)
 (:objects 
      shaker275 - shaker
      left right - hand
      shot212 shot496 - shot
      ingredient464 ingredient82 ingredient69 ingredient367 - ingredient
      cocktail389 - cocktail
      dispenser66 dispenser453 dispenser183 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker275)
  (ontable shot212)
  (ontable shot496)
  (dispenses dispenser66 ingredient464)
  (dispenses dispenser453 ingredient82)
  (dispenses dispenser183 ingredient69)
  (dispenses dispenser33 ingredient367)
  (clean shaker275)
  (clean shot212)
  (clean shot496)
  (empty shaker275)
  (empty shot212)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker275 l0)
  (shaker-level shaker275 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail389 ingredient69)
  (cocktail-part2 cocktail389 ingredient82)
)
 (:goal
  (and
      (contains shot212 cocktail389)
)))
