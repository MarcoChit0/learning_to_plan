; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 749701

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child37 - child
    bread335 - bread-portion
    content59 - content-portion
    tray440 tray313 tray403 - tray
    table454 table228 table268 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray440 kitchen)
     (at tray313 kitchen)
     (at tray403 kitchen)
     (at_kitchen_bread bread335)
     (at_kitchen_content content59)
     (not_allergic_gluten child37)
     (waiting child37 table268)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child37)
    )
  )
)
