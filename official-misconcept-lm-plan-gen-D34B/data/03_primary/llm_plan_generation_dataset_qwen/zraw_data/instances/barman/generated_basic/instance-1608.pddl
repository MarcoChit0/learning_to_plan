(define (problem prob)
 (:domain barman)
 (:objects 
      shaker396 - shaker
      left right - hand
      shot161 shot144 - shot
      ingredient426 ingredient49 ingredient414 - ingredient
      cocktail274 - cocktail
      dispenser210 dispenser165 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker396)
  (ontable shot161)
  (ontable shot144)
  (dispenses dispenser210 ingredient426)
  (dispenses dispenser165 ingredient49)
  (dispenses dispenser466 ingredient414)
  (clean shaker396)
  (clean shot161)
  (clean shot144)
  (empty shaker396)
  (empty shot161)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker396 l0)
  (shaker-level shaker396 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient414)
  (cocktail-part2 cocktail274 ingredient49)
)
 (:goal
  (and
      (contains shot161 cocktail274)
)))
