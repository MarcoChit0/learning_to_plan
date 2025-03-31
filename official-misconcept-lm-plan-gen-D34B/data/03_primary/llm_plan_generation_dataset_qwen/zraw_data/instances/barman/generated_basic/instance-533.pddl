(define (problem prob)
 (:domain barman)
 (:objects 
      shaker173 - shaker
      left right - hand
      shot154 shot419 - shot
      ingredient268 ingredient210 - ingredient
      cocktail1 - cocktail
      dispenser414 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker173)
  (ontable shot154)
  (ontable shot419)
  (dispenses dispenser414 ingredient268)
  (dispenses dispenser337 ingredient210)
  (clean shaker173)
  (clean shot154)
  (clean shot419)
  (empty shaker173)
  (empty shot154)
  (empty shot419)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker173 l0)
  (shaker-level shaker173 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient268)
  (cocktail-part2 cocktail1 ingredient210)
)
 (:goal
  (and
      (contains shot154 cocktail1)
)))
