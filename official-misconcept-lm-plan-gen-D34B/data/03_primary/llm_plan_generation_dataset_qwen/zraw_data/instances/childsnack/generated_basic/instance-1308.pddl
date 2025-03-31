; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 36862

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child248 - child
    bread21 bread480 - bread-portion
    content25 content191 - content-portion
    tray411 tray166 tray497 tray303 - tray
    table393 table395 table96 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray411 kitchen)
     (at tray166 kitchen)
     (at tray497 kitchen)
     (at tray303 kitchen)
     (at_kitchen_bread bread21)
     (at_kitchen_bread bread480)
     (at_kitchen_content content25)
     (at_kitchen_content content191)
     (no_gluten_bread bread480)
     (no_gluten_content content191)
     (allergic_gluten child248)
     (not_allergic_gluten child62)
     (waiting child62 table96)
     (waiting child248 table395)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child62)
     (served child248)
    )
  )
)
