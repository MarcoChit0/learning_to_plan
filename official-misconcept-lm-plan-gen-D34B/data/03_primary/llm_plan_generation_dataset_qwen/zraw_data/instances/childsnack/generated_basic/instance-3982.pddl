; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 97210

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child22 - child
    bread329 - bread-portion
    content185 - content-portion
    tray342 tray3 tray420 - tray
    table100 table265 table367 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray342 kitchen)
     (at tray3 kitchen)
     (at tray420 kitchen)
     (at_kitchen_bread bread329)
     (at_kitchen_content content185)
     (not_allergic_gluten child22)
     (waiting child22 table265)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child22)
    )
  )
)
