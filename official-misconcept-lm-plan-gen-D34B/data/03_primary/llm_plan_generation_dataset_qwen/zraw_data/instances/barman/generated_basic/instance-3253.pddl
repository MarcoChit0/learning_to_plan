(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot446 shot145 shot366 - shot
      ingredient380 ingredient200 ingredient239 ingredient314 - ingredient
      cocktail138 - cocktail
      dispenser71 dispenser288 dispenser213 dispenser208 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot446)
  (ontable shot145)
  (ontable shot366)
  (dispenses dispenser71 ingredient380)
  (dispenses dispenser288 ingredient200)
  (dispenses dispenser213 ingredient239)
  (dispenses dispenser208 ingredient314)
  (clean shaker143)
  (clean shot446)
  (clean shot145)
  (clean shot366)
  (empty shaker143)
  (empty shot446)
  (empty shot145)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail138 ingredient380)
  (cocktail-part2 cocktail138 ingredient200)
)
 (:goal
  (and
      (contains shot446 cocktail138)
      (contains shot145 ingredient200)
)))
