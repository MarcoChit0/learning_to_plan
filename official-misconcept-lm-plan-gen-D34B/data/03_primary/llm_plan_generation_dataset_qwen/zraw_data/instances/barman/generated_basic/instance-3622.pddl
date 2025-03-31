(define (problem prob)
 (:domain barman)
 (:objects 
      shaker417 - shaker
      left right - hand
      shot443 shot489 shot230 - shot
      ingredient365 ingredient284 ingredient315 - ingredient
      cocktail345 - cocktail
      dispenser344 dispenser37 dispenser229 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker417)
  (ontable shot443)
  (ontable shot489)
  (ontable shot230)
  (dispenses dispenser344 ingredient365)
  (dispenses dispenser37 ingredient284)
  (dispenses dispenser229 ingredient315)
  (clean shaker417)
  (clean shot443)
  (clean shot489)
  (clean shot230)
  (empty shaker417)
  (empty shot443)
  (empty shot489)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker417 l0)
  (shaker-level shaker417 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail345 ingredient365)
  (cocktail-part2 cocktail345 ingredient284)
)
 (:goal
  (and
      (contains shot443 cocktail345)
      (contains shot489 cocktail345)
)))
