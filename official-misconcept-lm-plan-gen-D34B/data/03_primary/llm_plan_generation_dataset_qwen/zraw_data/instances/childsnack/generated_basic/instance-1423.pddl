; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 106710

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child412 child35 - child
    bread127 bread418 - bread-portion
    content115 content490 - content-portion
    tray0 - tray
    table137 table311 table364 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray0 kitchen)
     (at_kitchen_bread bread127)
     (at_kitchen_bread bread418)
     (at_kitchen_content content115)
     (at_kitchen_content content490)
     (no_gluten_bread bread127)
     (no_gluten_content content115)
     (allergic_gluten child412)
     (not_allergic_gluten child35)
     (waiting child412 table364)
     (waiting child35 table311)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child412)
     (served child35)
    )
  )
)
