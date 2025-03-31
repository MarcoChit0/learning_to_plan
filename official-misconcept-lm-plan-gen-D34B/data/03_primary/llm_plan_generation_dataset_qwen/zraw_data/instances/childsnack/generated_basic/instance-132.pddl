; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 1568

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 child190 - child
    bread296 bread466 - bread-portion
    content26 content87 - content-portion
    tray268 tray211 tray403 tray44 - tray
    table42 table119 table191 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray268 kitchen)
     (at tray211 kitchen)
     (at tray403 kitchen)
     (at tray44 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread466)
     (at_kitchen_content content26)
     (at_kitchen_content content87)
     (not_allergic_gluten child190)
     (not_allergic_gluten child227)
     (waiting child227 table119)
     (waiting child190 table119)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child227)
     (served child190)
    )
  )
)
