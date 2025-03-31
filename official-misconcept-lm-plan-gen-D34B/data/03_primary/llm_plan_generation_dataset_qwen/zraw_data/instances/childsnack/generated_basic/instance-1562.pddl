; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 869778

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child135 child459 - child
    bread364 bread400 - bread-portion
    content16 content66 - content-portion
    tray471 tray132 tray176 tray20 - tray
    table242 table123 table55 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at tray132 kitchen)
     (at tray176 kitchen)
     (at tray20 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread400)
     (at_kitchen_content content16)
     (at_kitchen_content content66)
     (not_allergic_gluten child459)
     (not_allergic_gluten child135)
     (waiting child135 table123)
     (waiting child459 table242)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child135)
     (served child459)
    )
  )
)
