(define (problem prob)
 (:domain barman)
 (:objects 
      shaker11 - shaker
      left right - hand
      shot438 shot436 - shot
      ingredient32 ingredient245 - ingredient
      cocktail1 - cocktail
      dispenser32 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker11)
  (ontable shot438)
  (ontable shot436)
  (dispenses dispenser32 ingredient32)
  (dispenses dispenser420 ingredient245)
  (clean shaker11)
  (clean shot438)
  (clean shot436)
  (empty shaker11)
  (empty shot438)
  (empty shot436)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker11 l0)
  (shaker-level shaker11 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient32)
  (cocktail-part2 cocktail1 ingredient245)
)
 (:goal
  (and
      (contains shot438 cocktail1)
)))
