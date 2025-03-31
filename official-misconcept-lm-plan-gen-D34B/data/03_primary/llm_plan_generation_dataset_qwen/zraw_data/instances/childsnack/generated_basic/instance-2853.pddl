; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 803640

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child356 - child
    bread50 - bread-portion
    content487 - content-portion
    tray289 tray61 - tray
    table464 table434 table212 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at tray61 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_content content487)
     (not_allergic_gluten child356)
     (waiting child356 table464)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child356)
    )
  )
)
