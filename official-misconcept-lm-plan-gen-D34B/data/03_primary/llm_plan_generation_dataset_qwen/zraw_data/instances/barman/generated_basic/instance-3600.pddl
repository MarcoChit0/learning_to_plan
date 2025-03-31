(define (problem prob)
 (:domain barman)
 (:objects 
      shaker145 - shaker
      left right - hand
      shot143 shot65 shot43 shot452 - shot
      ingredient91 ingredient424 ingredient417 - ingredient
      cocktail289 - cocktail
      dispenser424 dispenser290 dispenser378 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker145)
  (ontable shot143)
  (ontable shot65)
  (ontable shot43)
  (ontable shot452)
  (dispenses dispenser424 ingredient91)
  (dispenses dispenser290 ingredient424)
  (dispenses dispenser378 ingredient417)
  (clean shaker145)
  (clean shot143)
  (clean shot65)
  (clean shot43)
  (clean shot452)
  (empty shaker145)
  (empty shot143)
  (empty shot65)
  (empty shot43)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker145 l0)
  (shaker-level shaker145 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail289 ingredient424)
  (cocktail-part2 cocktail289 ingredient91)
)
 (:goal
  (and
      (contains shot143 cocktail289)
      (contains shot65 ingredient417)
      (contains shot43 cocktail289)
)))
