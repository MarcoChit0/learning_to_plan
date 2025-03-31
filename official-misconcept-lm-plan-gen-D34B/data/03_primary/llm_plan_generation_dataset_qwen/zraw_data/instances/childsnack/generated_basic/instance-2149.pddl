; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 333696

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child273 child271 - child
    bread373 bread233 - bread-portion
    content78 content313 - content-portion
    tray344 tray79 tray63 - tray
    table302 table175 table266 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at tray79 kitchen)
     (at tray63 kitchen)
     (at_kitchen_bread bread373)
     (at_kitchen_bread bread233)
     (at_kitchen_content content78)
     (at_kitchen_content content313)
     (no_gluten_bread bread373)
     (no_gluten_content content78)
     (allergic_gluten child271)
     (not_allergic_gluten child273)
     (waiting child273 table266)
     (waiting child271 table266)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child273)
     (served child271)
    )
  )
)
