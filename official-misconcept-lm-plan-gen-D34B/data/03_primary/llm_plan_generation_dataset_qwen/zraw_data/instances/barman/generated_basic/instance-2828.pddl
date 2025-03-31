(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot166 shot410 shot29 - shot
      ingredient431 ingredient255 ingredient163 ingredient486 - ingredient
      cocktail225 - cocktail
      dispenser200 dispenser75 dispenser435 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot166)
  (ontable shot410)
  (ontable shot29)
  (dispenses dispenser200 ingredient431)
  (dispenses dispenser75 ingredient255)
  (dispenses dispenser435 ingredient163)
  (dispenses dispenser417 ingredient486)
  (clean shaker288)
  (clean shot166)
  (clean shot410)
  (clean shot29)
  (empty shaker288)
  (empty shot166)
  (empty shot410)
  (empty shot29)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail225 ingredient255)
  (cocktail-part2 cocktail225 ingredient163)
)
 (:goal
  (and
      (contains shot166 cocktail225)
      (contains shot410 cocktail225)
)))
