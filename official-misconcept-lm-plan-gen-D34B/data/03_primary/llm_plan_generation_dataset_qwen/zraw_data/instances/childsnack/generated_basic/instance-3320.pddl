; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 310103

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 child417 - child
    bread323 bread279 - bread-portion
    content448 content303 - content-portion
    tray450 tray492 tray209 tray321 - tray
    table46 table23 table80 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at tray492 kitchen)
     (at tray209 kitchen)
     (at tray321 kitchen)
     (at_kitchen_bread bread323)
     (at_kitchen_bread bread279)
     (at_kitchen_content content448)
     (at_kitchen_content content303)
     (no_gluten_bread bread323)
     (no_gluten_content content303)
     (allergic_gluten child417)
     (not_allergic_gluten child238)
     (waiting child238 table46)
     (waiting child417 table80)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child238)
     (served child417)
    )
  )
)
