; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 97611

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child219 - child
    bread273 - bread-portion
    content16 - content-portion
    tray59 - tray
    table274 table334 table429 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray59 kitchen)
     (at_kitchen_bread bread273)
     (at_kitchen_content content16)
     (not_allergic_gluten child219)
     (waiting child219 table429)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child219)
    )
  )
)
