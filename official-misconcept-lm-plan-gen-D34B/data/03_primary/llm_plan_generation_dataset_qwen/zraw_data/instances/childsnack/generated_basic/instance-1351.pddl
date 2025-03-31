; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 755881

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 child489 - child
    bread232 bread222 - bread-portion
    content384 content109 - content-portion
    tray463 tray329 tray459 - tray
    table58 table263 table100 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray463 kitchen)
     (at tray329 kitchen)
     (at tray459 kitchen)
     (at_kitchen_bread bread232)
     (at_kitchen_bread bread222)
     (at_kitchen_content content384)
     (at_kitchen_content content109)
     (not_allergic_gluten child187)
     (not_allergic_gluten child489)
     (waiting child187 table100)
     (waiting child489 table58)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child187)
     (served child489)
    )
  )
)
