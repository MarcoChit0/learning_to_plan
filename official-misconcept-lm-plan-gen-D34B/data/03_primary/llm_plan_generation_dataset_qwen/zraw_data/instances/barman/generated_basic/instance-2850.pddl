(define (problem prob)
 (:domain barman)
 (:objects 
      shaker134 - shaker
      left right - hand
      shot443 shot436 - shot
      ingredient165 ingredient412 ingredient191 ingredient58 - ingredient
      cocktail264 - cocktail
      dispenser40 dispenser48 dispenser18 dispenser219 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker134)
  (ontable shot443)
  (ontable shot436)
  (dispenses dispenser40 ingredient165)
  (dispenses dispenser48 ingredient412)
  (dispenses dispenser18 ingredient191)
  (dispenses dispenser219 ingredient58)
  (clean shaker134)
  (clean shot443)
  (clean shot436)
  (empty shaker134)
  (empty shot443)
  (empty shot436)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker134 l0)
  (shaker-level shaker134 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail264 ingredient165)
  (cocktail-part2 cocktail264 ingredient412)
)
 (:goal
  (and
      (contains shot443 cocktail264)
)))
