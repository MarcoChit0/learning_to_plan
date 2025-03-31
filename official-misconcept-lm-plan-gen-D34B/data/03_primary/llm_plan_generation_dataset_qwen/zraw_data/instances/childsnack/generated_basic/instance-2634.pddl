; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 611315

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child135 - child
    bread340 - bread-portion
    content16 - content-portion
    tray322 tray375 tray184 tray163 - tray
    table380 table350 table279 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray322 kitchen)
     (at tray375 kitchen)
     (at tray184 kitchen)
     (at tray163 kitchen)
     (at_kitchen_bread bread340)
     (at_kitchen_content content16)
     (not_allergic_gluten child135)
     (waiting child135 table380)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child135)
    )
  )
)
