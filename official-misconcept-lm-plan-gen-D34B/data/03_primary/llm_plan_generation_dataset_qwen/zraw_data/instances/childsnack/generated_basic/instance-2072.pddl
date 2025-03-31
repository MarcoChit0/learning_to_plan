; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 447192

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child384 - child
    bread67 - bread-portion
    content479 - content-portion
    tray475 tray132 tray464 tray381 - tray
    table266 table72 table245 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray475 kitchen)
     (at tray132 kitchen)
     (at tray464 kitchen)
     (at tray381 kitchen)
     (at_kitchen_bread bread67)
     (at_kitchen_content content479)
     (not_allergic_gluten child384)
     (waiting child384 table72)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child384)
    )
  )
)
