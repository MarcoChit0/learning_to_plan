(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot143 shot363 shot145 - shot
      ingredient6 ingredient289 ingredient229 ingredient52 - ingredient
      cocktail1 - cocktail
      dispenser308 dispenser334 dispenser174 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot143)
  (ontable shot363)
  (ontable shot145)
  (dispenses dispenser308 ingredient6)
  (dispenses dispenser334 ingredient289)
  (dispenses dispenser174 ingredient229)
  (dispenses dispenser310 ingredient52)
  (clean shaker350)
  (clean shot143)
  (clean shot363)
  (clean shot145)
  (empty shaker350)
  (empty shot143)
  (empty shot363)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient289)
  (cocktail-part2 cocktail1 ingredient229)
)
 (:goal
  (and
      (contains shot143 cocktail1)
      (contains shot363 ingredient6)
)))
