; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 548835

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child277 - child
    bread494 - bread-portion
    content262 - content-portion
    tray401 tray112 tray275 tray94 - tray
    table464 table228 table167 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray401 kitchen)
     (at tray112 kitchen)
     (at tray275 kitchen)
     (at tray94 kitchen)
     (at_kitchen_bread bread494)
     (at_kitchen_content content262)
     (not_allergic_gluten child277)
     (waiting child277 table464)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child277)
    )
  )
)
