(define (problem prob)
 (:domain barman)
 (:objects 
      shaker404 - shaker
      left right - hand
      shot155 shot42 - shot
      ingredient96 ingredient321 ingredient458 ingredient288 - ingredient
      cocktail39 - cocktail
      dispenser28 dispenser397 dispenser445 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker404)
  (ontable shot155)
  (ontable shot42)
  (dispenses dispenser28 ingredient96)
  (dispenses dispenser397 ingredient321)
  (dispenses dispenser445 ingredient458)
  (dispenses dispenser151 ingredient288)
  (clean shaker404)
  (clean shot155)
  (clean shot42)
  (empty shaker404)
  (empty shot155)
  (empty shot42)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker404 l0)
  (shaker-level shaker404 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient321)
  (cocktail-part2 cocktail39 ingredient458)
)
 (:goal
  (and
      (contains shot155 cocktail39)
)))
