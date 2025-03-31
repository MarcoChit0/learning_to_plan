; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 906128

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child13 - child
    bread313 - bread-portion
    content279 - content-portion
    tray197 tray475 tray12 tray469 - tray
    table278 table210 table27 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray197 kitchen)
     (at tray475 kitchen)
     (at tray12 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread313)
     (at_kitchen_content content279)
     (not_allergic_gluten child13)
     (waiting child13 table27)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child13)
    )
  )
)
