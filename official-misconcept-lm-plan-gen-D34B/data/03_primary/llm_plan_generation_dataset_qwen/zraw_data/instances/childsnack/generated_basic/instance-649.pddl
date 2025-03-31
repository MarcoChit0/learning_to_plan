; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 686089

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child51 child223 - child
    bread480 bread80 - bread-portion
    content377 content488 - content-portion
    tray234 tray121 - tray
    table6 table357 table47 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray234 kitchen)
     (at tray121 kitchen)
     (at_kitchen_bread bread480)
     (at_kitchen_bread bread80)
     (at_kitchen_content content377)
     (at_kitchen_content content488)
     (no_gluten_bread bread480)
     (no_gluten_content content488)
     (allergic_gluten child51)
     (not_allergic_gluten child223)
     (waiting child51 table47)
     (waiting child223 table6)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child51)
     (served child223)
    )
  )
)
