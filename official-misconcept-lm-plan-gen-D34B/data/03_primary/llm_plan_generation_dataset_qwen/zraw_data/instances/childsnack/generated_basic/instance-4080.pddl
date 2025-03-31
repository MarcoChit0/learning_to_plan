; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 301238

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child425 - child
    bread274 - bread-portion
    content183 - content-portion
    tray474 tray456 tray403 - tray
    table320 table397 table266 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray474 kitchen)
     (at tray456 kitchen)
     (at tray403 kitchen)
     (at_kitchen_bread bread274)
     (at_kitchen_content content183)
     (not_allergic_gluten child425)
     (waiting child425 table320)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child425)
    )
  )
)
