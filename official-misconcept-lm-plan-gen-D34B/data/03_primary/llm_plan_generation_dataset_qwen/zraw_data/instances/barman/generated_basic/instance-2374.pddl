(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot355 - shot
      ingredient226 ingredient48 ingredient143 ingredient401 - ingredient
      cocktail260 - cocktail
      dispenser365 dispenser259 dispenser318 dispenser56 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot355)
  (dispenses dispenser365 ingredient226)
  (dispenses dispenser259 ingredient48)
  (dispenses dispenser318 ingredient143)
  (dispenses dispenser56 ingredient401)
  (clean shaker181)
  (clean shot355)
  (empty shaker181)
  (empty shot355)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient226)
  (cocktail-part2 cocktail260 ingredient48)
)
 (:goal
  (and
      (contains shot355 cocktail260)
)))
