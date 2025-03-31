; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 124470

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 - child
    bread159 - bread-portion
    content55 - content-portion
    tray21 tray366 tray342 - tray
    table24 table247 table133 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at tray366 kitchen)
     (at tray342 kitchen)
     (at_kitchen_bread bread159)
     (at_kitchen_content content55)
     (not_allergic_gluten child167)
     (waiting child167 table24)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child167)
    )
  )
)
