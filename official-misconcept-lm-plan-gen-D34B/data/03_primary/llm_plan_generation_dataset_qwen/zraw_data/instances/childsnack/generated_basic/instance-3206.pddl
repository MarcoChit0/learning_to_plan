; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 735825

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child78 child18 - child
    bread263 bread344 - bread-portion
    content314 content363 - content-portion
    tray8 - tray
    table22 table442 table386 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at_kitchen_bread bread263)
     (at_kitchen_bread bread344)
     (at_kitchen_content content314)
     (at_kitchen_content content363)
     (no_gluten_bread bread263)
     (no_gluten_content content314)
     (allergic_gluten child78)
     (not_allergic_gluten child18)
     (waiting child78 table386)
     (waiting child18 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child78)
     (served child18)
    )
  )
)
