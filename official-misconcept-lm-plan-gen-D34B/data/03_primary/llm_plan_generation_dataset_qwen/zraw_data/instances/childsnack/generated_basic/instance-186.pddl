; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 280301

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child9 child237 - child
    bread388 bread17 - bread-portion
    content330 content437 - content-portion
    tray146 tray312 tray307 - tray
    table83 table170 table174 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray312 kitchen)
     (at tray307 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_bread bread17)
     (at_kitchen_content content330)
     (at_kitchen_content content437)
     (not_allergic_gluten child9)
     (not_allergic_gluten child237)
     (waiting child9 table174)
     (waiting child237 table174)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child9)
     (served child237)
    )
  )
)
