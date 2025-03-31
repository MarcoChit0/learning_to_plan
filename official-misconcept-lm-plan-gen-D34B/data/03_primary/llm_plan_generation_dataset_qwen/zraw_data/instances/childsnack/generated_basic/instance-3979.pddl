; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 766745

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child210 - child
    bread366 - bread-portion
    content17 - content-portion
    tray366 tray342 tray276 - tray
    table203 table354 table378 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray366 kitchen)
     (at tray342 kitchen)
     (at tray276 kitchen)
     (at_kitchen_bread bread366)
     (at_kitchen_content content17)
     (not_allergic_gluten child210)
     (waiting child210 table354)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child210)
    )
  )
)
