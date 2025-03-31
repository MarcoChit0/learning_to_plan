(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot444 shot143 shot150 - shot
      ingredient112 ingredient394 ingredient396 ingredient201 - ingredient
      cocktail225 - cocktail
      dispenser387 dispenser68 dispenser0 dispenser301 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot444)
  (ontable shot143)
  (ontable shot150)
  (dispenses dispenser387 ingredient112)
  (dispenses dispenser68 ingredient394)
  (dispenses dispenser0 ingredient396)
  (dispenses dispenser301 ingredient201)
  (clean shaker467)
  (clean shot444)
  (clean shot143)
  (clean shot150)
  (empty shaker467)
  (empty shot444)
  (empty shot143)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail225 ingredient396)
  (cocktail-part2 cocktail225 ingredient394)
)
 (:goal
  (and
      (contains shot444 cocktail225)
      (contains shot143 ingredient201)
)))
