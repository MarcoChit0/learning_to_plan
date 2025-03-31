; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 659734

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child287 - child
    bread53 - bread-portion
    content130 - content-portion
    tray493 tray117 tray49 - tray
    table230 table98 table14 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray493 kitchen)
     (at tray117 kitchen)
     (at tray49 kitchen)
     (at_kitchen_bread bread53)
     (at_kitchen_content content130)
     (not_allergic_gluten child287)
     (waiting child287 table230)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child287)
    )
  )
)
