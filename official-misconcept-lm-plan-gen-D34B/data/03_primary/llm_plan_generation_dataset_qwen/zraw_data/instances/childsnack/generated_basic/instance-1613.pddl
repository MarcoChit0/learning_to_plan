; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 471543

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child331 - child
    bread291 - bread-portion
    content181 - content-portion
    tray342 tray125 tray406 tray356 - tray
    table262 table411 table173 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray342 kitchen)
     (at tray125 kitchen)
     (at tray406 kitchen)
     (at tray356 kitchen)
     (at_kitchen_bread bread291)
     (at_kitchen_content content181)
     (not_allergic_gluten child331)
     (waiting child331 table173)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child331)
    )
  )
)
