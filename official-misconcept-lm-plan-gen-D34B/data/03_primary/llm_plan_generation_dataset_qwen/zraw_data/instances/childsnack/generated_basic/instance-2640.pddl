; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 424983

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 - child
    bread13 - bread-portion
    content127 - content-portion
    tray186 tray146 tray64 tray337 - tray
    table257 table291 table132 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray186 kitchen)
     (at tray146 kitchen)
     (at tray64 kitchen)
     (at tray337 kitchen)
     (at_kitchen_bread bread13)
     (at_kitchen_content content127)
     (not_allergic_gluten child55)
     (waiting child55 table257)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child55)
    )
  )
)
