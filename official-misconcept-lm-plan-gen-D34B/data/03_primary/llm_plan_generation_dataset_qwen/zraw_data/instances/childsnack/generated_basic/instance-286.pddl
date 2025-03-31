; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 760545

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 - child
    bread273 - bread-portion
    content331 - content-portion
    tray51 - tray
    table224 table319 table195 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray51 kitchen)
     (at_kitchen_bread bread273)
     (at_kitchen_content content331)
     (not_allergic_gluten child396)
     (waiting child396 table224)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child396)
    )
  )
)
