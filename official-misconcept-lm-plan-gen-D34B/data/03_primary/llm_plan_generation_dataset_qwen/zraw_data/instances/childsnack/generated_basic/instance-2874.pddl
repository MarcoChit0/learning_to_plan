; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 60369

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child383 - child
    bread274 - bread-portion
    content73 - content-portion
    tray188 tray147 - tray
    table484 table473 table75 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at tray147 kitchen)
     (at_kitchen_bread bread274)
     (at_kitchen_content content73)
     (not_allergic_gluten child383)
     (waiting child383 table473)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child383)
    )
  )
)
