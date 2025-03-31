(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot319 shot2 shot177 - shot
      ingredient222 ingredient181 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser481 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot319)
  (ontable shot2)
  (ontable shot177)
  (dispenses dispenser481 ingredient222)
  (dispenses dispenser290 ingredient181)
  (clean shaker215)
  (clean shot319)
  (clean shot2)
  (clean shot177)
  (empty shaker215)
  (empty shot319)
  (empty shot2)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient222)
  (cocktail-part2 cocktail1 ingredient181)
  (cocktail-part1 cocktail2 ingredient222)
  (cocktail-part2 cocktail2 ingredient181)
)
 (:goal
  (and
      (contains shot319 cocktail2)
      (contains shot2 cocktail1)
)))
