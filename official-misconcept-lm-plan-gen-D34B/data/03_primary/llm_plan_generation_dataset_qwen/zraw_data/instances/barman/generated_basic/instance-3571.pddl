(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot208 shot270 - shot
      ingredient61 ingredient318 ingredient64 ingredient82 - ingredient
      cocktail10 - cocktail
      dispenser16 dispenser290 dispenser118 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot208)
  (ontable shot270)
  (dispenses dispenser16 ingredient61)
  (dispenses dispenser290 ingredient318)
  (dispenses dispenser118 ingredient64)
  (dispenses dispenser345 ingredient82)
  (clean shaker286)
  (clean shot208)
  (clean shot270)
  (empty shaker286)
  (empty shot208)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail10 ingredient64)
  (cocktail-part2 cocktail10 ingredient82)
)
 (:goal
  (and
      (contains shot208 cocktail10)
)))
