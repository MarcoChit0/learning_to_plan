; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 894480

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child21 - child
    bread95 - bread-portion
    content337 - content-portion
    tray450 tray42 tray305 - tray
    table58 table343 table218 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at tray42 kitchen)
     (at tray305 kitchen)
     (at_kitchen_bread bread95)
     (at_kitchen_content content337)
     (not_allergic_gluten child21)
     (waiting child21 table58)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child21)
    )
  )
)
