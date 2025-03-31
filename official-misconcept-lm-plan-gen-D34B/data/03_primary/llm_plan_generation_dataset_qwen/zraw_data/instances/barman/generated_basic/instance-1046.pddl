(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot478 shot20 shot166 - shot
      ingredient278 ingredient331 ingredient148 ingredient168 - ingredient
      cocktail1 - cocktail
      dispenser293 dispenser468 dispenser84 dispenser25 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot478)
  (ontable shot20)
  (ontable shot166)
  (dispenses dispenser293 ingredient278)
  (dispenses dispenser468 ingredient331)
  (dispenses dispenser84 ingredient148)
  (dispenses dispenser25 ingredient168)
  (clean shaker459)
  (clean shot478)
  (clean shot20)
  (clean shot166)
  (empty shaker459)
  (empty shot478)
  (empty shot20)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient278)
  (cocktail-part2 cocktail1 ingredient168)
)
 (:goal
  (and
      (contains shot478 cocktail1)
      (contains shot20 ingredient278)
)))
