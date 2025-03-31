; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 18262

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child67 - child
    bread402 - bread-portion
    content213 - content-portion
    tray231 tray95 tray347 tray110 - tray
    table438 table294 table111 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray231 kitchen)
     (at tray95 kitchen)
     (at tray347 kitchen)
     (at tray110 kitchen)
     (at_kitchen_bread bread402)
     (at_kitchen_content content213)
     (not_allergic_gluten child67)
     (waiting child67 table111)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child67)
    )
  )
)
