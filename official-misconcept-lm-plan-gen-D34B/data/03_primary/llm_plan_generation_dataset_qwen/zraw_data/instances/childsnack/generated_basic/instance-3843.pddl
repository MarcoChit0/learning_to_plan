; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 631076

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child231 - child
    bread276 - bread-portion
    content496 - content-portion
    tray87 tray222 tray70 tray451 - tray
    table257 table344 table263 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray87 kitchen)
     (at tray222 kitchen)
     (at tray70 kitchen)
     (at tray451 kitchen)
     (at_kitchen_bread bread276)
     (at_kitchen_content content496)
     (not_allergic_gluten child231)
     (waiting child231 table263)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child231)
    )
  )
)
