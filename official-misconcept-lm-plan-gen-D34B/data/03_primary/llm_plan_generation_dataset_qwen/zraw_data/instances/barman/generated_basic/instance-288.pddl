(define (problem prob)
 (:domain barman)
 (:objects 
      shaker199 - shaker
      left right - hand
      shot4 shot138 - shot
      ingredient162 ingredient476 ingredient191 - ingredient
      cocktail1 - cocktail
      dispenser445 dispenser483 dispenser60 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker199)
  (ontable shot4)
  (ontable shot138)
  (dispenses dispenser445 ingredient162)
  (dispenses dispenser483 ingredient476)
  (dispenses dispenser60 ingredient191)
  (clean shaker199)
  (clean shot4)
  (clean shot138)
  (empty shaker199)
  (empty shot4)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker199 l0)
  (shaker-level shaker199 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient162)
  (cocktail-part2 cocktail1 ingredient191)
)
 (:goal
  (and
      (contains shot4 cocktail1)
)))
