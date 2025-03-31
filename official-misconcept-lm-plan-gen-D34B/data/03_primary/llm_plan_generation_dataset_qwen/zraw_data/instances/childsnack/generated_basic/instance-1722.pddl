; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 551958

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 - child
    bread446 - bread-portion
    content348 - content-portion
    tray369 tray277 tray129 - tray
    table468 table170 table298 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at tray277 kitchen)
     (at tray129 kitchen)
     (at_kitchen_bread bread446)
     (at_kitchen_content content348)
     (not_allergic_gluten child170)
     (waiting child170 table170)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child170)
    )
  )
)
