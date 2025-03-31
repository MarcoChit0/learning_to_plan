; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 985268

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 child277 - child
    bread122 bread92 - bread-portion
    content89 content203 - content-portion
    tray70 tray387 - tray
    table338 table230 table87 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray387 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_bread bread92)
     (at_kitchen_content content89)
     (at_kitchen_content content203)
     (no_gluten_bread bread122)
     (no_gluten_content content89)
     (allergic_gluten child277)
     (not_allergic_gluten child92)
     (waiting child92 table338)
     (waiting child277 table230)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child92)
     (served child277)
    )
  )
)
