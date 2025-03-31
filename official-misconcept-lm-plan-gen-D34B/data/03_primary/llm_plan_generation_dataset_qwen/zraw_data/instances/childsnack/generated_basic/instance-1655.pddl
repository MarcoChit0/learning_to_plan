; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 227205

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child272 - child
    bread299 - bread-portion
    content243 - content-portion
    tray8 tray488 tray213 tray91 - tray
    table441 table276 table397 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at tray488 kitchen)
     (at tray213 kitchen)
     (at tray91 kitchen)
     (at_kitchen_bread bread299)
     (at_kitchen_content content243)
     (not_allergic_gluten child272)
     (waiting child272 table441)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child272)
    )
  )
)
