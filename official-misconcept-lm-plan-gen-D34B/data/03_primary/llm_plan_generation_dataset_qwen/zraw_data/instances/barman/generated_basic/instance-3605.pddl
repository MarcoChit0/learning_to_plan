(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot440 shot411 shot288 shot459 - shot
      ingredient342 ingredient390 ingredient366 - ingredient
      cocktail30 - cocktail
      dispenser60 dispenser259 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot440)
  (ontable shot411)
  (ontable shot288)
  (ontable shot459)
  (dispenses dispenser60 ingredient342)
  (dispenses dispenser259 ingredient390)
  (dispenses dispenser346 ingredient366)
  (clean shaker72)
  (clean shot440)
  (clean shot411)
  (clean shot288)
  (clean shot459)
  (empty shaker72)
  (empty shot440)
  (empty shot411)
  (empty shot288)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient342)
  (cocktail-part2 cocktail30 ingredient390)
)
 (:goal
  (and
      (contains shot440 cocktail30)
      (contains shot411 cocktail30)
      (contains shot288 ingredient390)
)))
