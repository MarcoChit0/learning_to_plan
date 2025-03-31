(define (problem prob)
 (:domain barman)
 (:objects 
      shaker240 - shaker
      left right - hand
      shot442 shot258 shot145 - shot
      ingredient460 ingredient406 - ingredient
      cocktail212 - cocktail
      dispenser32 dispenser354 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker240)
  (ontable shot442)
  (ontable shot258)
  (ontable shot145)
  (dispenses dispenser32 ingredient460)
  (dispenses dispenser354 ingredient406)
  (clean shaker240)
  (clean shot442)
  (clean shot258)
  (clean shot145)
  (empty shaker240)
  (empty shot442)
  (empty shot258)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker240 l0)
  (shaker-level shaker240 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail212 ingredient406)
  (cocktail-part2 cocktail212 ingredient460)
)
 (:goal
  (and
      (contains shot442 cocktail212)
      (contains shot258 cocktail212)
)))
