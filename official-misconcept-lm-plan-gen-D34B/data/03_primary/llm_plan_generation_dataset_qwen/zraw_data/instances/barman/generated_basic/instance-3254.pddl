(define (problem prob)
 (:domain barman)
 (:objects 
      shaker391 - shaker
      left right - hand
      shot85 shot143 shot318 - shot
      ingredient367 ingredient271 ingredient213 ingredient263 - ingredient
      cocktail361 - cocktail
      dispenser94 dispenser449 dispenser370 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker391)
  (ontable shot85)
  (ontable shot143)
  (ontable shot318)
  (dispenses dispenser94 ingredient367)
  (dispenses dispenser449 ingredient271)
  (dispenses dispenser370 ingredient213)
  (dispenses dispenser397 ingredient263)
  (clean shaker391)
  (clean shot85)
  (clean shot143)
  (clean shot318)
  (empty shaker391)
  (empty shot85)
  (empty shot143)
  (empty shot318)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker391 l0)
  (shaker-level shaker391 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail361 ingredient367)
  (cocktail-part2 cocktail361 ingredient213)
)
 (:goal
  (and
      (contains shot85 cocktail361)
      (contains shot143 ingredient213)
)))
