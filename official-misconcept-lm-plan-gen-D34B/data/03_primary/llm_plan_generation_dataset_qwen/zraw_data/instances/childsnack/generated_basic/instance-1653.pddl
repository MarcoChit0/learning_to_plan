; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 911307

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child67 - child
    bread359 - bread-portion
    content497 - content-portion
    tray416 tray197 tray238 tray325 - tray
    table347 table318 table447 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray416 kitchen)
     (at tray197 kitchen)
     (at tray238 kitchen)
     (at tray325 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_content content497)
     (not_allergic_gluten child67)
     (waiting child67 table447)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child67)
    )
  )
)
